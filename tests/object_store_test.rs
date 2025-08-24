use lyra::stdlib::io::object_store::{
    object_store_open, object_store_read, object_store_write, object_store_delete, object_store_head, object_store_list,
};
use lyra::vm::Value;
use std::collections::HashMap;

fn assoc(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = HashMap::new();
    for (k, v) in pairs { m.insert(k.to_string(), v); }
    Value::Object(m)
}

fn s(bytes: &str) -> Value { Value::String(bytes.into()) }

#[test]
fn test_memory_store_rw_head_list_delete() {
    // Open memory store
    let store = object_store_open(&[assoc(vec![
        ("provider", s("memory")),
        ("root", s("ns1")),
    ])]).unwrap();

    // Write key
    assert_eq!(
        object_store_write(&[Value::List(vec![store.clone(), s("k1")]), s("hello")]).unwrap(),
        Value::Boolean(true)
    );
    // Head size
    let h = object_store_head(&[Value::List(vec![store.clone(), s("k1")])]).unwrap();
    if let Value::Object(m) = h { assert_eq!(m.get("size"), Some(&Value::Integer(5))); } else { panic!("expected head object"); }
    // Read
    let v = object_store_read(&[Value::List(vec![store.clone(), s("k1")])]).unwrap();
    if let Value::List(bytes) = v { assert_eq!(bytes.len(), 5); } else { panic!("expected bytes list"); }
    // List prefix ""
    let l = object_store_list(&[Value::List(vec![store.clone(), s("")])]).unwrap();
    if let Value::List(names) = l { assert!(names.iter().any(|n| matches!(n, Value::String(s) if s == "k1"))); } else { panic!("expected list"); }
    // Delete
    assert_eq!(object_store_delete(&[Value::List(vec![store.clone(), s("k1")])]).unwrap(), Value::Boolean(true));
    // After delete, head returns Missing
    assert_eq!(object_store_head(&[Value::List(vec![store.clone(), s("k1")])]).unwrap(), Value::Missing);
}

#[test]
fn test_file_store_with_tempdir() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_string_lossy().to_string();
    let store = object_store_open(&[assoc(vec![
        ("provider", s("file")),
        ("root", Value::String(root.clone())),
    ])]).unwrap();

    // Write to subpath
    assert_eq!(
        object_store_write(&[Value::List(vec![store.clone(), s("sub/foo.txt")]), s("abc")]).unwrap(),
        Value::Boolean(true)
    );
    // Head size 3
    let h = object_store_head(&[Value::List(vec![store.clone(), s("sub/foo.txt")])]).unwrap();
    if let Value::Object(m) = h { assert_eq!(m.get("size"), Some(&Value::Integer(3))); } else { panic!("expected head object"); }

    // Read via URI path as well
    let uri = Value::String(format!("file://{}/sub/foo.txt", root));
    let v = object_store_read(&[uri]).unwrap();
    if let Value::List(bytes) = v { assert_eq!(bytes.len(), 3); } else { panic!("expected bytes list"); }

    // List under sub should see foo.txt
    let l = object_store_list(&[Value::List(vec![store.clone(), s("sub")])]).unwrap();
    if let Value::List(names) = l { assert!(names.iter().any(|n| matches!(n, Value::String(s) if s == "foo.txt"))); } else { panic!("expected list"); }

    // Delete using {store, key}
    assert_eq!(object_store_delete(&[Value::List(vec![store.clone(), s("sub/foo.txt")])]).unwrap(), Value::Boolean(true));
}

