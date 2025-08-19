use lyra::serialization::{BinaryWriter, BinaryReader, Serializable};
use lyra::ast::{Number, Symbol};
use std::io::Cursor;

#[test]
fn test_basic_binary_writer() {
    let mut buffer = Vec::new();
    {
        let mut writer = BinaryWriter::new(&mut buffer);
        writer.write_bytes(b"hello").unwrap();
        writer.flush().unwrap();
    }
    assert_eq!(buffer, b"hello");
}

#[test]
fn test_basic_binary_reader() {
    let data = b"hello";
    let mut reader = BinaryReader::new(Cursor::new(data));
    
    let mut result = vec![0u8; 5];
    reader.read_bytes(&mut result).unwrap();
    assert_eq!(result, b"hello");
}

#[test]
fn test_string_serialization() {
    let original = "Hello, World!".to_string();
    
    let mut buffer = Vec::new();
    original.serialize(&mut buffer).unwrap();
    
    let mut reader = Cursor::new(buffer);
    let deserialized = String::deserialize(&mut reader).unwrap();
    
    assert_eq!(original, deserialized);
}

#[test]
fn test_number_serialization() {
    let int_num = Number::Integer(42);
    let real_num = Number::Real(3.14159);
    
    // Test integer
    let mut buffer = Vec::new();
    int_num.serialize(&mut buffer).unwrap();
    let mut reader = Cursor::new(buffer);
    let deserialized_int = Number::deserialize(&mut reader).unwrap();
    assert_eq!(int_num, deserialized_int);
    
    // Test real
    let mut buffer = Vec::new();
    real_num.serialize(&mut buffer).unwrap();
    let mut reader = Cursor::new(buffer);
    let deserialized_real = Number::deserialize(&mut reader).unwrap();
    assert_eq!(real_num, deserialized_real);
}

#[test]
fn test_symbol_serialization() {
    let symbol = Symbol { name: "TestSymbol".to_string() };
    
    let mut buffer = Vec::new();
    symbol.serialize(&mut buffer).unwrap();
    
    let mut reader = Cursor::new(buffer);
    let deserialized = Symbol::deserialize(&mut reader).unwrap();
    
    assert_eq!(symbol, deserialized);
}