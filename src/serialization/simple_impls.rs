//! Simple serialization implementations for core types

use super::{Serializable, SerializationResult, BinaryWriter, BinaryReader};
use crate::ast::{Number, Symbol};
use std::io::{Read, Write};

// Helper functions
fn write_string<W: Write>(writer: &mut BinaryWriter<W>, s: &str) -> SerializationResult<()> {
    let bytes = s.as_bytes();
    writer.write_bytes(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_bytes(bytes)?;
    Ok(())
}

fn read_string<R: Read>(reader: &mut BinaryReader<R>) -> SerializationResult<String> {
    let mut len_bytes = [0u8; 4];
    reader.read_bytes(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    
    let mut bytes = vec![0u8; len];
    reader.read_bytes(&mut bytes)?;
    
    String::from_utf8(bytes).map_err(|_| super::SerializationError::CorruptData)
}

impl Serializable for String {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        write_string(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        read_string(&mut reader)
    }
    
    fn serialized_size(&self) -> usize {
        4 + self.len()
    }
}

impl Serializable for Symbol {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        write_string(&mut writer, &self.name)?;
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let name = read_string(&mut reader)?;
        Ok(Symbol { name })
    }
    
    fn serialized_size(&self) -> usize {
        4 + self.name.len()
    }
}

impl Serializable for Number {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            Number::Integer(i) => {
                writer.write_bytes(&[0u8])?; // Integer tag
                writer.write_bytes(&i.to_le_bytes())?;
            }
            Number::Real(f) => {
                writer.write_bytes(&[1u8])?; // Real tag
                writer.write_bytes(&f.to_le_bytes())?;
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
                Ok(Number::Integer(i64::from_le_bytes(bytes)))
            }
            1 => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(Number::Real(f64::from_le_bytes(bytes)))
            }
            _ => Err(super::SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        1 + 8 // tag + 8 bytes for i64/f64
    }
}