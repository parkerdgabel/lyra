//! Zero-copy deserialization with memory mapping

use super::{SerializationError, SerializationResult, LYRA_MAGIC, SERIALIZATION_VERSION};
use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::marker::PhantomData;

/// Memory-mapped file for zero-copy operations
pub struct MappedFile {
    _file: File,
    mmap: Mmap,
}

impl MappedFile {
    pub fn open<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { 
            Mmap::map(&file).map_err(|e| {
                SerializationError::MemoryMappingError(format!("Failed to map file: {}", e))
            })?
        };
        
        Ok(Self {
            _file: file,
            mmap,
        })
    }
    
    pub fn create<P: AsRef<Path>>(path: P, size: usize) -> SerializationResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        
        file.set_len(size as u64)?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| {
                    SerializationError::MemoryMappingError(format!("Failed to map file: {}", e))
                })?
                .make_read_only()
                .map_err(|e| {
                    SerializationError::MemoryMappingError(format!("Failed to make read-only: {}", e))
                })?
        };
        
        Ok(Self {
            _file: file,
            mmap,
        })
    }
    
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }
    
    pub fn len(&self) -> usize {
        self.mmap.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

/// Zero-copy deserializer for memory-mapped data
pub struct ZeroCopyDeserializer<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> ZeroCopyDeserializer<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
        }
    }
    
    pub fn from_mapped_file(file: &'a MappedFile) -> SerializationResult<Self> {
        let data = file.data();
        
        // Validate magic number
        if data.len() < LYRA_MAGIC.len() {
            return Err(SerializationError::CorruptData);
        }
        
        if &data[0..LYRA_MAGIC.len()] != LYRA_MAGIC {
            return Err(SerializationError::CorruptData);
        }
        
        // Validate version
        if data.len() < LYRA_MAGIC.len() + 4 {
            return Err(SerializationError::CorruptData);
        }
        
        let version_bytes = &data[LYRA_MAGIC.len()..LYRA_MAGIC.len() + 4];
        let version = u32::from_le_bytes([
            version_bytes[0], version_bytes[1], 
            version_bytes[2], version_bytes[3]
        ]);
        
        if version != SERIALIZATION_VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: SERIALIZATION_VERSION,
                actual: version,
            });
        }
        
        // Skip magic and version for actual data
        Ok(Self {
            data: &data[LYRA_MAGIC.len() + 4..],
            position: 0,
        })
    }
    
    pub fn read_slice(&mut self, len: usize) -> SerializationResult<&'a [u8]> {
        if self.position + len > self.data.len() {
            return Err(SerializationError::CorruptData);
        }
        
        let slice = &self.data[self.position..self.position + len];
        self.position += len;
        Ok(slice)
    }
    
    pub fn read_u32(&mut self) -> SerializationResult<u32> {
        let bytes = self.read_slice(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    
    pub fn read_u64(&mut self) -> SerializationResult<u64> {
        let bytes = self.read_slice(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
    
    pub fn read_string(&mut self) -> SerializationResult<&'a str> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_slice(len)?;
        std::str::from_utf8(bytes).map_err(|_| SerializationError::CorruptData)
    }
    
    pub fn position(&self) -> usize {
        self.position
    }
    
    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }
    
    pub fn reset(&mut self) {
        self.position = 0;
    }
    
    pub fn seek(&mut self, position: usize) -> SerializationResult<()> {
        if position > self.data.len() {
            return Err(SerializationError::CorruptData);
        }
        self.position = position;
        Ok(())
    }
}

/// Zero-copy reference to serialized data that doesn't require deserialization
pub struct ZeroCopyRef<'a, T> {
    data: &'a [u8],
    _phantom: PhantomData<T>,
}

impl<'a, T> ZeroCopyRef<'a, T> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }
    
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Lazy loading container for large data structures
pub struct LazyLoader<'a, T> {
    deserializer: ZeroCopyDeserializer<'a>,
    loaded: Option<T>,
    _phantom: PhantomData<T>,
}

impl<'a, T> LazyLoader<'a, T> {
    pub fn new(deserializer: ZeroCopyDeserializer<'a>) -> Self {
        Self {
            deserializer,
            loaded: None,
            _phantom: PhantomData,
        }
    }
    
    pub fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }
    
    pub fn size_hint(&self) -> usize {
        self.deserializer.remaining()
    }
}

impl<'a, T> LazyLoader<'a, T>
where
    T: super::Serializable,
{
    pub fn load(&mut self) -> SerializationResult<&T> {
        if self.loaded.is_none() {
            // This would require a way to deserialize from ZeroCopyDeserializer
            // For now, we'll return an error indicating this needs to be implemented
            // when we have the full Serializable trait working with zero-copy
            return Err(SerializationError::UnsupportedFormat(
                "Zero-copy deserialization not yet implemented for this type".to_string()
            ));
        }
        
        Ok(self.loaded.as_ref().unwrap())
    }
}

/// Streaming deserializer for processing large files chunk by chunk
pub struct StreamingDeserializer<'a> {
    data: &'a [u8],
    chunk_size: usize,
    position: usize,
}

impl<'a> StreamingDeserializer<'a> {
    pub fn new(data: &'a [u8], chunk_size: usize) -> Self {
        Self {
            data,
            chunk_size,
            position: 0,
        }
    }
    
    pub fn next_chunk(&mut self) -> Option<&'a [u8]> {
        if self.position >= self.data.len() {
            return None;
        }
        
        let end = (self.position + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.position..end];
        self.position = end;
        
        Some(chunk)
    }
    
    pub fn remaining_chunks(&self) -> usize {
        if self.position >= self.data.len() {
            0
        } else {
            (self.data.len() - self.position + self.chunk_size - 1) / self.chunk_size
        }
    }
    
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_zero_copy_deserializer() {
        let data = b"Hello, World!";
        let mut deserializer = ZeroCopyDeserializer::new(data);
        
        let slice = deserializer.read_slice(5).unwrap();
        assert_eq!(slice, b"Hello");
        
        assert_eq!(deserializer.remaining(), 8);
        assert_eq!(deserializer.position(), 5);
    }
    
    #[test]
    fn test_zero_copy_deserializer_numbers() {
        let mut data = Vec::new();
        data.extend_from_slice(&42u32.to_le_bytes());
        data.extend_from_slice(&1234567890u64.to_le_bytes());
        
        let mut deserializer = ZeroCopyDeserializer::new(&data);
        
        let num1 = deserializer.read_u32().unwrap();
        assert_eq!(num1, 42);
        
        let num2 = deserializer.read_u64().unwrap();
        assert_eq!(num2, 1234567890);
    }
    
    #[test]
    fn test_zero_copy_deserializer_string() {
        let mut data = Vec::new();
        let test_str = "Hello, World!";
        data.extend_from_slice(&(test_str.len() as u32).to_le_bytes());
        data.extend_from_slice(test_str.as_bytes());
        
        let mut deserializer = ZeroCopyDeserializer::new(&data);
        let result = deserializer.read_string().unwrap();
        assert_eq!(result, test_str);
    }
    
    #[test]
    fn test_mapped_file() -> std::io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        let test_data = b"This is test data for memory mapping";
        temp_file.write_all(test_data)?;
        temp_file.flush()?;
        
        let mapped = MappedFile::open(temp_file.path()).unwrap();
        assert_eq!(mapped.data(), test_data);
        assert_eq!(mapped.len(), test_data.len());
        assert!(!mapped.is_empty());
        
        Ok(())
    }
    
    #[test]
    fn test_mapped_file_with_format() -> std::io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        
        // Write magic number and version
        temp_file.write_all(LYRA_MAGIC)?;
        temp_file.write_all(&SERIALIZATION_VERSION.to_le_bytes())?;
        temp_file.write_all(b"payload data")?;
        temp_file.flush()?;
        
        let mapped = MappedFile::open(temp_file.path()).unwrap();
        let deserializer = ZeroCopyDeserializer::from_mapped_file(&mapped).unwrap();
        
        assert_eq!(deserializer.remaining(), 12); // "payload data".len()
        
        Ok(())
    }
    
    #[test]
    fn test_streaming_deserializer() {
        let data = b"This is a long string for testing streaming deserialization";
        let mut streamer = StreamingDeserializer::new(data, 10);
        
        let chunk1 = streamer.next_chunk().unwrap();
        assert_eq!(chunk1, b"This is a ");
        
        let chunk2 = streamer.next_chunk().unwrap();
        assert_eq!(chunk2, b"long strin");
        
        assert!(streamer.remaining_chunks() > 0);
        
        // Reset and try again
        streamer.reset();
        let first_chunk_again = streamer.next_chunk().unwrap();
        assert_eq!(first_chunk_again, b"This is a ");
    }
    
    #[test]
    fn test_zero_copy_ref() {
        let data = b"reference data";
        let zero_copy_ref = ZeroCopyRef::<String>::new(data);
        
        assert_eq!(zero_copy_ref.data(), data);
        assert_eq!(zero_copy_ref.len(), data.len());
        assert!(!zero_copy_ref.is_empty());
    }
}