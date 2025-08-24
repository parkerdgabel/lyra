use lyra::stdlib::io::object_store::{object_store_open, object_store_write, object_store_read};
use lyra::vm::Value;
use std::collections::HashMap;

fn assoc(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = HashMap::new();
    for (k, v) in pairs { m.insert(k.to_string(), v); }
    Value::Object(m)
}

fn s(x: &str) -> Value { Value::String(x.into()) }

#[test]
fn test_object_store_stream_reader_memory() {
    let store = object_store_open(&[assoc(vec![("provider", s("memory")), ("root", s("ns"))])]).unwrap();
    object_store_write(&[Value::List(vec![store.clone(), s("k")]), s("abcdef")]).unwrap();
    let reader = object_store_read(&[Value::List(vec![store.clone(), s("k")]), assoc(vec![("stream", Value::Boolean(true))])]).unwrap();
    if let Value::LyObj(obj) = reader {
        // Read first 3 bytes
        let chunk = obj.call_method("Read", &[Value::Integer(3)]).unwrap();
        if let Value::List(bytes) = chunk { assert_eq!(bytes.len(), 3); } else { panic!("expected list of bytes"); }
        // BytesRead should be 3
        let pos = obj.call_method("BytesRead", &[]).unwrap();
        assert_eq!(pos, Value::Integer(3));
    } else { panic!("expected reader object"); }
}

