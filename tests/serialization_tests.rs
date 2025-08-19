use lyra::ast::{Expr, Number, Pattern};
use lyra::serialization::{Serializable, BinaryWriter, BinaryReader, LYRA_MAGIC, SERIALIZATION_VERSION};
use lyra::vm::Value;
use std::io::Cursor;

#[test]
fn test_expr_serialization_roundtrip() {
    let expr = Expr::function(
        Expr::symbol("Plus"), 
        vec![Expr::integer(1), Expr::integer(2)]
    );
    
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    expr.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized = Expr::deserialize(&mut reader).unwrap();
    
    assert_eq!(expr, deserialized);
}

#[test]
fn test_value_serialization_roundtrip() {
    let value = Value::List(vec![
        Value::Integer(42),
        Value::Real(3.14),
        Value::String("hello".to_string()),
        Value::Boolean(true),
    ]);
    
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    value.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized = Value::deserialize(&mut reader).unwrap();
    
    assert_eq!(value, deserialized);
}

#[test]
fn test_number_serialization() {
    let int_num = Number::Integer(42);
    let real_num = Number::Real(3.14159);
    
    // Test integer
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    int_num.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized_int = Number::deserialize(&mut reader).unwrap();
    assert_eq!(int_num, deserialized_int);
    
    // Test real
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    real_num.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized_real = Number::deserialize(&mut reader).unwrap();
    assert_eq!(real_num, deserialized_real);
}

#[test]
fn test_pattern_serialization() {
    let pattern = Pattern::Named {
        name: "x".to_string(),
        pattern: Box::new(Pattern::Blank { head: Some("Integer".to_string()) }),
    };
    
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    pattern.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized = Pattern::deserialize(&mut reader).unwrap();
    
    assert_eq!(pattern, deserialized);
}

#[test]
fn test_serialization_with_magic_and_version() {
    let expr = Expr::symbol("test");
    
    let mut buffer = Vec::new();
    
    // Write magic number and version
    buffer.extend_from_slice(LYRA_MAGIC);
    buffer.extend_from_slice(&SERIALIZATION_VERSION.to_le_bytes());
    
    let mut writer = BinaryWriter::new(&mut buffer);
    expr.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    // Verify magic and version
    assert_eq!(&buffer[0..8], LYRA_MAGIC);
    let version = u32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
    assert_eq!(version, SERIALIZATION_VERSION);
    
    // Deserialize the expression
    let mut reader = BinaryReader::new(Cursor::new(&buffer[12..]));
    let deserialized = Expr::deserialize(&mut reader).unwrap();
    assert_eq!(expr, deserialized);
}

#[test]
fn test_large_expression_serialization() {
    // Create a deeply nested expression
    let mut expr = Expr::integer(1);
    for i in 2..=100 {
        expr = Expr::function(
            Expr::symbol("Plus"),
            vec![expr, Expr::integer(i)]
        );
    }
    
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    expr.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized = Expr::deserialize(&mut reader).unwrap();
    
    assert_eq!(expr, deserialized);
}

#[test]
fn test_serialized_size_estimation() {
    let expr = Expr::function(
        Expr::symbol("List"), 
        vec![Expr::integer(1), Expr::integer(2), Expr::integer(3)]
    );
    
    let estimated_size = expr.serialized_size();
    
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    expr.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    // The estimate should be reasonably close to actual size
    let actual_size = buffer.len();
    assert!(estimated_size > 0);
    // Allow some variance but should be in the right ballpark
    assert!(estimated_size <= actual_size * 2);
}

#[test]
fn test_empty_structures() {
    let empty_list = Expr::list(vec![]);
    
    let mut buffer = Vec::new();
    let mut writer = BinaryWriter::new(&mut buffer);
    empty_list.serialize(&mut writer).unwrap();
    writer.flush().unwrap();
    
    let mut reader = BinaryReader::new(Cursor::new(buffer));
    let deserialized = Expr::deserialize(&mut reader).unwrap();
    
    assert_eq!(empty_list, deserialized);
}