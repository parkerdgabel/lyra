//! Session persistence for REPL state

use super::{
    SerializationError, SerializationResult, Serializable, BinaryWriter, BinaryReader,
    LYRA_MAGIC, SERIALIZATION_VERSION
};
use super::compression::{CompressionConfig, CompressedData, SmartCompressor};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// For now, we'll use a simple Value type to avoid VM dependency issues
// In practice, this would use crate::vm::Value
#[derive(Debug, Clone, PartialEq)]
pub enum SimpleValue {
    Integer(i64),
    Real(f64),
    String(String),
    Symbol(String),
    List(Vec<SimpleValue>),
    Boolean(bool),
    Missing,
}

impl Serializable for SimpleValue {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            SimpleValue::Integer(i) => {
                writer.write_bytes(&[0u8])?;
                writer.write_bytes(&i.to_le_bytes())?;
            }
            SimpleValue::Real(f) => {
                writer.write_bytes(&[1u8])?;
                writer.write_bytes(&f.to_le_bytes())?;
            }
            SimpleValue::String(s) => {
                writer.write_bytes(&[2u8])?;
                writer.write_bytes(&(s.len() as u32).to_le_bytes())?;
                writer.write_bytes(s.as_bytes())?;
            }
            SimpleValue::Symbol(s) => {
                writer.write_bytes(&[3u8])?;
                writer.write_bytes(&(s.len() as u32).to_le_bytes())?;
                writer.write_bytes(s.as_bytes())?;
            }
            SimpleValue::List(items) => {
                writer.write_bytes(&[4u8])?;
                writer.write_bytes(&(items.len() as u32).to_le_bytes())?;
                for item in items {
                    item.serialize(&mut writer)?;
                }
            }
            SimpleValue::Boolean(b) => {
                writer.write_bytes(&[5u8])?;
                writer.write_bytes(&[if *b { 1u8 } else { 0u8 }])?;
            }
            SimpleValue::Missing => {
                writer.write_bytes(&[6u8])?;
            }
        }
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let mut tag = [0u8; 1];
        reader.read_bytes(&mut tag)?;
        
        match tag[0] {
            0 => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(SimpleValue::Integer(i64::from_le_bytes(bytes)))
            }
            1 => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(SimpleValue::Real(f64::from_le_bytes(bytes)))
            }
            2 => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                let mut bytes = vec![0u8; len];
                reader.read_bytes(&mut bytes)?;
                let s = String::from_utf8(bytes).map_err(|_| SerializationError::CorruptData)?;
                Ok(SimpleValue::String(s))
            }
            3 => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                let mut bytes = vec![0u8; len];
                reader.read_bytes(&mut bytes)?;
                let s = String::from_utf8(bytes).map_err(|_| SerializationError::CorruptData)?;
                Ok(SimpleValue::Symbol(s))
            }
            4 => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(SimpleValue::deserialize(&mut reader)?);
                }
                Ok(SimpleValue::List(items))
            }
            5 => {
                let mut byte = [0u8; 1];
                reader.read_bytes(&mut byte)?;
                Ok(SimpleValue::Boolean(byte[0] == 1))
            }
            6 => Ok(SimpleValue::Missing),
            _ => Err(SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        match self {
            SimpleValue::Integer(_) => 1 + 8,
            SimpleValue::Real(_) => 1 + 8,
            SimpleValue::String(s) => 1 + 4 + s.len(),
            SimpleValue::Symbol(s) => 1 + 4 + s.len(),
            SimpleValue::List(items) => 1 + 4 + items.iter().map(|i| i.serialized_size()).sum::<usize>(),
            SimpleValue::Boolean(_) => 1 + 1,
            SimpleValue::Missing => 1,
        }
    }
}

/// REPL session state with metadata
#[derive(Debug, Clone)]
pub struct SessionState {
    pub variables: HashMap<String, SimpleValue>,
    pub history: Vec<String>,
    pub working_directory: PathBuf,
    pub timestamp: u64,
    pub version: u32,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            history: Vec::new(),
            working_directory: std::env::current_dir().unwrap_or_default(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            version: SERIALIZATION_VERSION,
        }
    }
}

impl SessionState {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_variable(&mut self, name: String, value: SimpleValue) {
        self.variables.insert(name, value);
        self.update_timestamp();
    }
    
    pub fn add_history_entry(&mut self, command: String) {
        self.history.push(command);
        // Keep history to reasonable size
        if self.history.len() > 10000 {
            self.history.drain(0..self.history.len() - 10000);
        }
        self.update_timestamp();
    }
    
    pub fn clear_variables(&mut self) {
        self.variables.clear();
        self.update_timestamp();
    }
    
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.update_timestamp();
    }
    
    fn update_timestamp(&mut self) {
        self.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

impl Serializable for SessionState {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        
        // Write timestamp
        writer.write_bytes(&self.timestamp.to_le_bytes())?;
        
        // Write version
        writer.write_bytes(&self.version.to_le_bytes())?;
        
        // Write working directory
        let wd_str = self.working_directory.to_string_lossy();
        writer.write_bytes(&(wd_str.len() as u32).to_le_bytes())?;
        writer.write_bytes(wd_str.as_bytes())?;
        
        // Write variables
        writer.write_bytes(&(self.variables.len() as u32).to_le_bytes())?;
        for (name, value) in &self.variables {
            writer.write_bytes(&(name.len() as u32).to_le_bytes())?;
            writer.write_bytes(name.as_bytes())?;
            value.serialize(&mut writer)?;
        }
        
        // Write history
        writer.write_bytes(&(self.history.len() as u32).to_le_bytes())?;
        for entry in &self.history {
            writer.write_bytes(&(entry.len() as u32).to_le_bytes())?;
            writer.write_bytes(entry.as_bytes())?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        
        // Read timestamp
        let mut timestamp_bytes = [0u8; 8];
        reader.read_bytes(&mut timestamp_bytes)?;
        let timestamp = u64::from_le_bytes(timestamp_bytes);
        
        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_bytes(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        
        // Read working directory
        let mut wd_len_bytes = [0u8; 4];
        reader.read_bytes(&mut wd_len_bytes)?;
        let wd_len = u32::from_le_bytes(wd_len_bytes) as usize;
        let mut wd_bytes = vec![0u8; wd_len];
        reader.read_bytes(&mut wd_bytes)?;
        let wd_str = String::from_utf8(wd_bytes).map_err(|_| SerializationError::CorruptData)?;
        let working_directory = PathBuf::from(wd_str);
        
        // Read variables
        let mut vars_len_bytes = [0u8; 4];
        reader.read_bytes(&mut vars_len_bytes)?;
        let vars_len = u32::from_le_bytes(vars_len_bytes) as usize;
        let mut variables = HashMap::with_capacity(vars_len);
        
        for _ in 0..vars_len {
            let mut name_len_bytes = [0u8; 4];
            reader.read_bytes(&mut name_len_bytes)?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;
            let mut name_bytes = vec![0u8; name_len];
            reader.read_bytes(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes).map_err(|_| SerializationError::CorruptData)?;
            
            let value = SimpleValue::deserialize(&mut reader)?;
            variables.insert(name, value);
        }
        
        // Read history
        let mut hist_len_bytes = [0u8; 4];
        reader.read_bytes(&mut hist_len_bytes)?;
        let hist_len = u32::from_le_bytes(hist_len_bytes) as usize;
        let mut history = Vec::with_capacity(hist_len);
        
        for _ in 0..hist_len {
            let mut entry_len_bytes = [0u8; 4];
            reader.read_bytes(&mut entry_len_bytes)?;
            let entry_len = u32::from_le_bytes(entry_len_bytes) as usize;
            let mut entry_bytes = vec![0u8; entry_len];
            reader.read_bytes(&mut entry_bytes)?;
            let entry = String::from_utf8(entry_bytes).map_err(|_| SerializationError::CorruptData)?;
            history.push(entry);
        }
        
        Ok(SessionState {
            variables,
            history,
            working_directory,
            timestamp,
            version,
        })
    }
    
    fn serialized_size(&self) -> usize {
        let mut size = 8 + 4; // timestamp + version
        size += 4 + self.working_directory.to_string_lossy().len(); // working directory
        size += 4; // variables count
        for (name, value) in &self.variables {
            size += 4 + name.len() + value.serialized_size();
        }
        size += 4; // history count
        for entry in &self.history {
            size += 4 + entry.len();
        }
        size
    }
}

/// Configuration for session management
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub session_file: PathBuf,
    pub auto_save: bool,
    pub auto_save_interval: u64, // seconds
    pub compression: bool,
    pub backup_count: usize,
    pub max_history_size: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_file: PathBuf::from(".lyra_session"),
            auto_save: true,
            auto_save_interval: 60,
            compression: true,
            backup_count: 5,
            max_history_size: 10000,
        }
    }
}

/// Manages session persistence for the REPL with advanced features
pub struct SessionManager {
    config: SessionConfig,
    compressor: SmartCompressor,
    last_save_time: u64,
}

impl SessionManager {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            compressor: SmartCompressor::new(),
            last_save_time: 0,
        }
    }
    
    pub fn with_file<P: AsRef<Path>>(session_file: P) -> Self {
        let mut config = SessionConfig::default();
        config.session_file = session_file.as_ref().to_path_buf();
        Self::new(config)
    }
    
    /// Save session state to file with compression and error handling
    pub fn save_session(&mut self, state: &SessionState) -> SerializationResult<()> {
        // Create backup of existing session file
        if self.config.session_file.exists() {
            self.create_backup()?;
        }
        
        // Serialize session state
        let mut buffer = Vec::new();
        
        // Write magic number and version
        buffer.extend_from_slice(LYRA_MAGIC);
        buffer.extend_from_slice(&SERIALIZATION_VERSION.to_le_bytes());
        
        // Serialize the state
        state.serialize(&mut buffer)?;
        
        // Compress if enabled
        let final_data = if self.config.compression {
            let compressed = self.compressor.compress_adaptive(&buffer)?;
            
            // Write compression header (algorithm + original size)
            let mut compressed_buffer = Vec::new();
            compressed_buffer.push(compressed.algorithm as u8);
            compressed_buffer.extend_from_slice(&compressed.original_size.to_le_bytes());
            compressed_buffer.extend_from_slice(&compressed.compressed_data);
            compressed_buffer
        } else {
            // Write no compression marker
            let mut uncompressed_buffer = Vec::new();
            uncompressed_buffer.push(0u8); // No compression
            uncompressed_buffer.extend_from_slice(&buffer.len().to_le_bytes());
            uncompressed_buffer.extend_from_slice(&buffer);
            uncompressed_buffer
        };
        
        // Write to file atomically
        let temp_file = self.config.session_file.with_extension("tmp");
        {
            let mut file = File::create(&temp_file)?;
            file.write_all(&final_data)?;
            file.sync_all()?;
        }
        
        // Atomic move
        std::fs::rename(&temp_file, &self.config.session_file)?;
        
        self.last_save_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Ok(())
    }
    
    /// Load session state from file with decompression and validation
    pub fn load_session(&self) -> SerializationResult<SessionState> {
        if !self.config.session_file.exists() {
            return Ok(SessionState::default());
        }
        
        let mut file = File::open(&self.config.session_file)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        if buffer.len() < LYRA_MAGIC.len() + 4 {
            return Err(SerializationError::CorruptData);
        }
        
        // Validate magic number
        if &buffer[0..LYRA_MAGIC.len()] != LYRA_MAGIC {
            return Err(SerializationError::CorruptData);
        }
        
        // Validate version
        let version_offset = LYRA_MAGIC.len();
        let version = u32::from_le_bytes([
            buffer[version_offset],
            buffer[version_offset + 1],
            buffer[version_offset + 2],
            buffer[version_offset + 3],
        ]);
        
        if version != SERIALIZATION_VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: SERIALIZATION_VERSION,
                actual: version,
            });
        }
        
        // Skip magic and version
        let data_offset = LYRA_MAGIC.len() + 4;
        let compressed_data = &buffer[data_offset..];
        
        if compressed_data.is_empty() {
            return Err(SerializationError::CorruptData);
        }
        
        // Check compression
        let algorithm = compressed_data[0];
        let original_size_bytes = &compressed_data[1..9];
        let original_size = usize::from_le_bytes([
            original_size_bytes[0], original_size_bytes[1],
            original_size_bytes[2], original_size_bytes[3],
            original_size_bytes[4], original_size_bytes[5],
            original_size_bytes[6], original_size_bytes[7],
        ]);
        
        let decompressed_data = if algorithm == 0 {
            // No compression
            compressed_data[9..].to_vec()
        } else {
            // Decompress
            use super::compression::{decompress, CompressionAlgorithm};
            let compression_algo = match algorithm {
                1 => CompressionAlgorithm::Lz4,
                2 => CompressionAlgorithm::Zstd,
                _ => return Err(SerializationError::UnsupportedFormat(
                    format!("Unknown compression algorithm: {}", algorithm)
                )),
            };
            decompress(&compressed_data[9..], compression_algo)?
        };
        
        // Validate decompressed size
        if decompressed_data.len() != original_size {
            return Err(SerializationError::CorruptData);
        }
        
        // Deserialize
        let mut cursor = std::io::Cursor::new(decompressed_data);
        SessionState::deserialize(&mut cursor)
    }
    
    /// Check if auto-save is due
    pub fn should_auto_save(&self) -> bool {
        if !self.config.auto_save {
            return false;
        }
        
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        current_time - self.last_save_time >= self.config.auto_save_interval
    }
    
    /// Create backup of existing session file
    fn create_backup(&self) -> SerializationResult<()> {
        if !self.config.session_file.exists() {
            return Ok(());
        }
        
        // Create backup with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let backup_name = format!(
            "{}.backup.{}",
            self.config.session_file.to_string_lossy(),
            timestamp
        );
        let backup_path = self.config.session_file.with_file_name(backup_name);
        
        std::fs::copy(&self.config.session_file, &backup_path)?;
        
        // Clean up old backups
        self.cleanup_old_backups()?;
        
        Ok(())
    }
    
    /// Remove old backup files, keeping only the configured number
    fn cleanup_old_backups(&self) -> SerializationResult<()> {
        let parent_dir = self.config.session_file.parent().unwrap_or(Path::new("."));
        let session_name = self.config.session_file.file_name().unwrap().to_string_lossy();
        
        let mut backups = Vec::new();
        
        for entry in std::fs::read_dir(parent_dir)? {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().to_string();
            
            if file_name.starts_with(&format!("{}.backup.", session_name)) {
                backups.push((file_name, entry.path()));
            }
        }
        
        // Sort by timestamp (newest first)
        backups.sort_by(|a, b| b.0.cmp(&a.0));
        
        // Remove old backups
        for (_, path) in backups.into_iter().skip(self.config.backup_count) {
            let _ = std::fs::remove_file(path); // Ignore errors
        }
        
        Ok(())
    }
    
    /// Get session file information
    pub fn session_info(&self) -> SerializationResult<SessionInfo> {
        if !self.config.session_file.exists() {
            return Ok(SessionInfo {
                exists: false,
                size: 0,
                modified: 0,
                compressed: false,
            });
        }
        
        let metadata = std::fs::metadata(&self.config.session_file)?;
        let modified = metadata.modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // Quick check for compression by reading the header
        let mut file = File::open(&self.config.session_file)?;
        let mut header = vec![0u8; LYRA_MAGIC.len() + 4 + 1];
        let bytes_read = file.read(&mut header)?;
        
        let compressed = if bytes_read >= header.len() {
            header[LYRA_MAGIC.len() + 4] != 0 // compression algorithm != 0
        } else {
            false
        };
        
        Ok(SessionInfo {
            exists: true,
            size: metadata.len(),
            modified,
            compressed,
        })
    }
}

/// Information about a session file
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub exists: bool,
    pub size: u64,
    pub modified: u64,
    pub compressed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_simple_value_serialization() {
        let values = vec![
            SimpleValue::Integer(42),
            SimpleValue::Real(3.14),
            SimpleValue::String("Hello".to_string()),
            SimpleValue::Symbol("test".to_string()),
            SimpleValue::Boolean(true),
            SimpleValue::Missing,
            SimpleValue::List(vec![
                SimpleValue::Integer(1),
                SimpleValue::Integer(2),
                SimpleValue::Integer(3),
            ]),
        ];
        
        for value in values {
            let mut buffer = Vec::new();
            value.serialize(&mut buffer).unwrap();
            
            let mut cursor = std::io::Cursor::new(buffer);
            let deserialized = SimpleValue::deserialize(&mut cursor).unwrap();
            
            assert_eq!(value, deserialized);
        }
    }
    
    #[test]
    fn test_session_state_serialization() {
        let mut state = SessionState::new();
        state.add_variable("x".to_string(), SimpleValue::Integer(42));
        state.add_variable("y".to_string(), SimpleValue::String("hello".to_string()));
        state.add_history_entry("x = 42".to_string());
        state.add_history_entry("y = \"hello\"".to_string());
        
        let mut buffer = Vec::new();
        state.serialize(&mut buffer).unwrap();
        
        let mut cursor = std::io::Cursor::new(buffer);
        let deserialized = SessionState::deserialize(&mut cursor).unwrap();
        
        assert_eq!(state.variables, deserialized.variables);
        assert_eq!(state.history, deserialized.history);
        assert_eq!(state.working_directory, deserialized.working_directory);
    }
    
    #[test]
    fn test_session_manager() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = SessionConfig {
            session_file: temp_file.path().to_path_buf(),
            auto_save: true,
            auto_save_interval: 1,
            compression: true,
            backup_count: 2,
            max_history_size: 100,
        };
        
        let mut manager = SessionManager::new(config);
        
        // Create test session
        let mut state = SessionState::new();
        state.add_variable("test".to_string(), SimpleValue::Integer(123));
        state.add_history_entry("test = 123".to_string());
        
        // Save session
        manager.save_session(&state).unwrap();
        
        // Load session
        let loaded_state = manager.load_session().unwrap();
        
        assert_eq!(state.variables, loaded_state.variables);
        assert_eq!(state.history, loaded_state.history);
        
        // Check session info
        let info = manager.session_info().unwrap();
        assert!(info.exists);
        assert!(info.size > 0);
        assert!(info.compressed);
    }
    
    #[test]
    fn test_session_manager_no_compression() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = SessionConfig {
            session_file: temp_file.path().to_path_buf(),
            compression: false,
            ..Default::default()
        };
        
        let mut manager = SessionManager::new(config);
        
        let mut state = SessionState::new();
        state.add_variable("x".to_string(), SimpleValue::Integer(42));
        
        manager.save_session(&state).unwrap();
        let loaded_state = manager.load_session().unwrap();
        
        assert_eq!(state.variables, loaded_state.variables);
        
        let info = manager.session_info().unwrap();
        assert!(!info.compressed);
    }
}