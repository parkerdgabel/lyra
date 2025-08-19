//! High-performance binary serialization system for Lyra symbolic computation engine
//!
//! This module provides:
//! - Binary serialization with version awareness
//! - Zero-copy deserialization with memory mapping
//! - Session persistence for REPL state
//! - Compression and optimization for large datasets
//! - Cross-platform compatibility

use thiserror::Error;
use std::io::{Read, Write};

pub mod compression;
pub mod platform;
pub mod session;
pub mod simple_impls;
pub mod zero_copy;

// Re-export simple implementations
pub use simple_impls::*;

/// Serialization errors
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
    #[error("Corruption detected in serialized data")]
    CorruptData,
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
    #[error("Memory mapping error: {0}")]
    MemoryMappingError(String),
    #[error("Cross-platform compatibility error: {0}")]
    PlatformError(String),
}

pub type SerializationResult<T> = Result<T, SerializationError>;

/// Current serialization format version
pub const SERIALIZATION_VERSION: u32 = 1;

/// Binary format magic number to identify Lyra serialized data
pub const LYRA_MAGIC: &[u8; 8] = b"LYRA\x01\x02\x03\x04";

/// Core trait for serializable types in Lyra
pub trait Serializable: Sized {
    /// Serialize to binary format with version information
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()>;
    
    /// Deserialize from binary format with version validation
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self>;
    
    /// Get the estimated serialized size in bytes
    fn serialized_size(&self) -> usize;
    
    /// Check if this type supports zero-copy deserialization
    fn supports_zero_copy() -> bool {
        false
    }
}

/// High-performance binary writer with buffering
pub struct BinaryWriter<W: Write> {
    writer: W,
    buffer: Vec<u8>,
    position: usize,
}

impl<W: Write> BinaryWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(8192), // 8KB buffer
            position: 0,
        }
    }
    
    pub fn with_capacity(writer: W, capacity: usize) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(capacity),
            position: 0,
        }
    }
    
    pub fn flush(&mut self) -> SerializationResult<()> {
        if self.position > 0 {
            self.writer.write_all(&self.buffer[..self.position])?;
            self.position = 0;
        }
        self.writer.flush()?;
        Ok(())
    }
    
    pub fn write_bytes(&mut self, data: &[u8]) -> SerializationResult<()> {
        if self.position + data.len() > self.buffer.capacity() {
            self.flush()?;
        }
        
        if data.len() > self.buffer.capacity() {
            // Write large data directly
            self.writer.write_all(data)?;
        } else {
            // Resize buffer if needed and copy data
            if self.buffer.len() < self.position + data.len() {
                self.buffer.resize(self.position + data.len(), 0);
            }
            self.buffer[self.position..self.position + data.len()].copy_from_slice(data);
            self.position += data.len();
        }
        Ok(())
    }
}

impl<W: Write> Drop for BinaryWriter<W> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// High-performance binary reader with buffering
pub struct BinaryReader<R: Read> {
    reader: R,
    buffer: Vec<u8>,
    position: usize,
    end: usize,
}

impl<R: Read> BinaryReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: vec![0; 8192], // 8KB buffer
            position: 0,
            end: 0,
        }
    }
    
    pub fn with_capacity(reader: R, capacity: usize) -> Self {
        Self {
            reader,
            buffer: vec![0; capacity],
            position: 0,
            end: 0,
        }
    }
    
    pub fn read_bytes(&mut self, data: &mut [u8]) -> SerializationResult<()> {
        let mut remaining = data.len();
        let mut offset = 0;
        
        while remaining > 0 {
            if self.position >= self.end {
                // Refill buffer
                self.position = 0;
                self.end = self.reader.read(&mut self.buffer)?;
                if self.end == 0 {
                    return Err(SerializationError::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Unexpected end of stream"
                    )));
                }
            }
            
            let available = self.end - self.position;
            let to_copy = remaining.min(available);
            
            data[offset..offset + to_copy].copy_from_slice(
                &self.buffer[self.position..self.position + to_copy]
            );
            
            self.position += to_copy;
            offset += to_copy;
            remaining -= to_copy;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    
    #[test]
    fn test_binary_writer_basic() {
        let mut buffer = Vec::new();
        {
            let mut writer = BinaryWriter::new(&mut buffer);
            writer.write_bytes(b"hello").unwrap();
            writer.write_bytes(b"world").unwrap();
            writer.flush().unwrap();
        } // writer is dropped here, releasing the borrow
        
        assert_eq!(buffer, b"helloworld");
    }
    
    #[test]
    fn test_binary_writer_large_data() {
        let mut buffer = Vec::new();
        let large_data = vec![42u8; 1000];
        {
            let mut writer = BinaryWriter::with_capacity(&mut buffer, 16);
            writer.write_bytes(&large_data).unwrap();
            writer.flush().unwrap();
        } // writer is dropped here, releasing the borrow
        
        assert_eq!(buffer, large_data);
    }
    
    #[test]
    fn test_binary_reader_basic() {
        let data = b"helloworld";
        let mut reader = BinaryReader::new(Cursor::new(data));
        
        let mut result = vec![0u8; 5];
        reader.read_bytes(&mut result).unwrap();
        assert_eq!(result, b"hello");
        
        let mut result = vec![0u8; 5];
        reader.read_bytes(&mut result).unwrap();
        assert_eq!(result, b"world");
    }
    
    #[test]
    fn test_binary_reader_large_data() {
        let data = vec![42u8; 1000];
        let mut reader = BinaryReader::with_capacity(Cursor::new(&data), 16);
        
        let mut result = vec![0u8; 1000];
        reader.read_bytes(&mut result).unwrap();
        assert_eq!(result, data);
    }
    
    #[test]
    fn test_binary_reader_eof() {
        let data = b"hello";
        let mut reader = BinaryReader::new(Cursor::new(data));
        
        let mut result = vec![0u8; 10]; // Try to read more than available
        let error = reader.read_bytes(&mut result).unwrap_err();
        
        match error {
            SerializationError::Io(io_error) => {
                assert_eq!(io_error.kind(), std::io::ErrorKind::UnexpectedEof);
            }
            _ => panic!("Expected IO error"),
        }
    }
    
    #[test]
    fn test_magic_number() {
        assert_eq!(LYRA_MAGIC, b"LYRA\x01\x02\x03\x04");
    }
    
    #[test]
    fn test_version() {
        assert_eq!(SERIALIZATION_VERSION, 1);
    }
}