//! Cross-platform compatibility for serialization

use super::{SerializationError, SerializationResult};
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write};

/// Platform-specific serialization utilities
pub struct PlatformWriter<W: Write> {
    writer: W,
}

impl<W: Write> PlatformWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
    
    pub fn write_u32(&mut self, value: u32) -> SerializationResult<()> {
        self.writer.write_u32::<LittleEndian>(value)?;
        Ok(())
    }
    
    pub fn write_u64(&mut self, value: u64) -> SerializationResult<()> {
        self.writer.write_u64::<LittleEndian>(value)?;
        Ok(())
    }
    
    pub fn write_f64(&mut self, value: f64) -> SerializationResult<()> {
        self.writer.write_f64::<LittleEndian>(value)?;
        Ok(())
    }
}

pub struct PlatformReader<R: Read> {
    reader: R,
}

impl<R: Read> PlatformReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
    
    pub fn read_u32(&mut self) -> SerializationResult<u32> {
        Ok(self.reader.read_u32::<LittleEndian>()?)
    }
    
    pub fn read_u64(&mut self) -> SerializationResult<u64> {
        Ok(self.reader.read_u64::<LittleEndian>()?)
    }
    
    pub fn read_f64(&mut self) -> SerializationResult<f64> {
        Ok(self.reader.read_f64::<LittleEndian>()?)
    }
}