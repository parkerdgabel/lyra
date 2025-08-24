//! Serialization utilities (YAML/TOML/CBOR/MessagePack/Bincode)

use crate::vm::{Value, VmError, VmResult};
use std::collections::HashMap;

// -------- Conversion helpers between serde values and VM Value --------

fn value_from_serde_json(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Missing,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Integer(i) }
            else if let Some(f) = n.as_f64() { Value::Real(f) }
            else { Value::Real(0.0) }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(value_from_serde_json).collect()),
        serde_json::Value::Object(map) => {
            let mut out = HashMap::new();
            for (k, v) in map { out.insert(k.clone(), value_from_serde_json(v)); }
            Value::Object(out)
        }
    }
}

fn value_to_serde_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Missing => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::Integer(i) => serde_json::Value::Number((*i).into()),
        Value::Real(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_serde_json).collect()),
        Value::Object(map) => {
            let mut m = serde_json::Map::new();
            for (k, v) in map { m.insert(k.clone(), value_to_serde_json(v)); }
            serde_json::Value::Object(m)
        }
        other => serde_json::Value::String(format!("{:?}", other)),
    }
}

// YAML
pub fn yaml_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 {
        return Err(VmError::TypeError { expected: "YAMLParse[input, opts?]".to_string(), actual: format!("{} args", args.len()) });
    }
    // Accept String or List[byte]
    let input = match &args[0] {
        Value::String(s) => s.clone(),
        Value::List(items) => {
            let mut buf = Vec::with_capacity(items.len());
            for it in items { match it { Value::Integer(i) if *i >= 0 && *i <= 255 => buf.push(*i as u8), _ => return Err(VmError::TypeError { expected: "List of byte integers 0..255".to_string(), actual: format!("{:?}", it) }) } }
            String::from_utf8(buf).map_err(|e| VmError::Runtime(format!("Invalid UTF-8: {}", e)))?
        }
        v => return Err(VmError::TypeError { expected: "String or Bytes".to_string(), actual: format!("{:?}", v) })
    };
    let y: serde_yaml::Value = serde_yaml::from_str(&input)
        .map_err(|e| VmError::Runtime(format!("YAML parse error: {}", e)))?;
    // Convert via serde_json bridge to reuse converters
    let j = serde_json::to_value(y).map_err(|e| VmError::Runtime(format!("YAML->JSON convert error: {}", e)))?;
    Ok(value_from_serde_json(&j))
}

pub fn yaml_stringify(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 {
        return Err(VmError::TypeError { expected: "YAMLStringify[expr, opts?]".to_string(), actual: format!("{} args", args.len()) });
    }
    let j = value_to_serde_json(&args[0]);
    let yv: serde_yaml::Value = serde_yaml::to_value(j).map_err(|e| VmError::Runtime(format!("JSON->YAML convert error: {}", e)))?;
    let s = serde_yaml::to_string(&yv).map_err(|e| VmError::Runtime(format!("YAML stringify error: {}", e)))?;
    Ok(Value::String(s))
}

// TOML
pub fn toml_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 { return Err(VmError::TypeError { expected: "TOMLParse[string|bytes]".to_string(), actual: format!("{} args", args.len()) }); }
    let input = match &args[0] {
        Value::String(s) => s.clone(),
        Value::List(items) => {
            let mut buf = Vec::with_capacity(items.len());
            for it in items { match it { Value::Integer(i) if *i >= 0 && *i <= 255 => buf.push(*i as u8), _ => return Err(VmError::TypeError { expected: "List of byte integers 0..255".to_string(), actual: format!("{:?}", it) }) } }
            String::from_utf8(buf).map_err(|e| VmError::Runtime(format!("Invalid UTF-8: {}", e)))?
        }
        v => return Err(VmError::TypeError { expected: "String or Bytes".to_string(), actual: format!("{:?}", v) })
    };
    let tv: toml::Value = input.parse::<toml::Value>()
        .map_err(|e| VmError::Runtime(format!("TOML parse error: {}", e)))?;
    let j = toml_to_json(&tv);
    Ok(value_from_serde_json(&j))
}

pub fn toml_stringify(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "TOMLStringify[assoc, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let j = value_to_serde_json(&args[0]);
    let tv = json_to_toml(&j).map_err(|e| VmError::Runtime(format!("JSON->TOML convert error: {}", e)))?;
    let s = toml::to_string(&tv).map_err(|e| VmError::Runtime(format!("TOML stringify error: {}", e)))?;
    Ok(Value::String(s))
}

fn toml_to_json(tv: &toml::Value) -> serde_json::Value {
    match tv {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
        toml::Value::Array(arr) => serde_json::Value::Array(arr.iter().map(toml_to_json).collect()),
        toml::Value::Table(map) => {
            let mut m = serde_json::Map::new();
            for (k, v) in map { m.insert(k.clone(), toml_to_json(v)); }
            serde_json::Value::Object(m)
        }
    }
}

fn json_to_toml(j: &serde_json::Value) -> Result<toml::Value, String> {
    match j {
        serde_json::Value::Null => Ok(toml::Value::String("null".into())),
        serde_json::Value::Bool(b) => Ok(toml::Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Ok(toml::Value::Integer(i)) }
            else if let Some(f) = n.as_f64() { Ok(toml::Value::Float(f)) }
            else { Err("Unsupported number".into()) }
        }
        serde_json::Value::String(s) => Ok(toml::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let mut out = Vec::new();
            for v in arr { out.push(json_to_toml(v)?); }
            Ok(toml::Value::Array(out))
        }
        serde_json::Value::Object(map) => {
            let mut out = toml::map::Map::new();
            for (k, v) in map { out.insert(k.clone(), json_to_toml(v)?); }
            Ok(toml::Value::Table(out))
        }
    }
}

pub fn cbor_read(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "CBORRead[input, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    Err(VmError::Runtime("CBORRead is not implemented in this phase".to_string()))
}

pub fn cbor_write(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "CBORWrite[expr, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    Err(VmError::Runtime("CBORWrite is not implemented in this phase".to_string()))
}

pub fn msgpack_read(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "MessagePackRead[input, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    Err(VmError::Runtime("MessagePackRead is not implemented in this phase".to_string()))
}

pub fn msgpack_write(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "MessagePackWrite[expr, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    Err(VmError::Runtime("MessagePackWrite is not implemented in this phase".to_string()))
}

pub fn bincode_read(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "BincodeRead[input, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    Err(VmError::Runtime("BincodeRead is not implemented in this phase".to_string()))
}

pub fn bincode_write(args: &[Value]) -> VmResult<Value> {
    if args.len() == 0 || args.len() > 2 { return Err(VmError::TypeError { expected: "BincodeWrite[expr, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    Err(VmError::Runtime("BincodeWrite is not implemented in this phase".to_string()))
}

/// Registration helper for serialization functions
pub fn register_serialization_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = std::collections::HashMap::new();
    f.insert("YAMLParse".to_string(), yaml_parse as fn(&[Value]) -> VmResult<Value>);
    f.insert("YAMLStringify".to_string(), yaml_stringify as fn(&[Value]) -> VmResult<Value>);
    f.insert("TOMLParse".to_string(), toml_parse as fn(&[Value]) -> VmResult<Value>);
    f.insert("TOMLStringify".to_string(), toml_stringify as fn(&[Value]) -> VmResult<Value>);
    f.insert("CBORRead".to_string(), cbor_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("CBORWrite".to_string(), cbor_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("MessagePackRead".to_string(), msgpack_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("MessagePackWrite".to_string(), msgpack_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("BincodeRead".to_string(), bincode_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("BincodeWrite".to_string(), bincode_write as fn(&[Value]) -> VmResult<Value>);
    f
}
