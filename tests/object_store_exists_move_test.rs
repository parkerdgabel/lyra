use lyra::stdlib::io::object_store as os;
use lyra::vm::Value;
use std::collections::HashMap;

#[test]
fn test_memory_exists_and_move() {
    // Create memory store
    let mut m = HashMap::new();
    m.insert("provider".to_string(), Value::String("memory".to_string()));
    m.insert("root".to_string(), Value::String("ns".to_string()));
    let store = os::object_store_open(&[Value::Object(m)]).unwrap();

    // Write value
    let key = Value::String("a".to_string());
    let data = Value::String("hello".to_string());
    let _ = os::object_store_write(&[Value::List(vec![store.clone(), key.clone()]), data]).unwrap();

    // Exists true
    let ex = os::object_store_exists(&[Value::List(vec![store.clone(), key.clone()])]).unwrap();
    assert_eq!(ex, Value::Boolean(true));

    // Move
    let dst = Value::List(vec![store.clone(), Value::String("b".to_string())]);
    let _ = os::object_store_move(&[Value::List(vec![store.clone(), key.clone()]), dst.clone()]).unwrap();

    // Old does not exist, new exists
    let ex_old = os::object_store_exists(&[Value::List(vec![store.clone(), key.clone()])]).unwrap();
    let ex_new = os::object_store_exists(&[dst]).unwrap();
    assert_eq!(ex_old, Value::Boolean(false));
    assert_eq!(ex_new, Value::Boolean(true));
}

#[test]
fn test_file_exists_and_move() {
    let dir = tempfile::tempdir().unwrap();
    let p1 = dir.path().join("x.txt");
    let p2 = dir.path().join("y.txt");
    let uri1 = format!("file://{}", p1.to_string_lossy());
    let uri2 = format!("file://{}", p2.to_string_lossy());
    let data = Value::String("data".to_string());
    os::object_store_write(&[Value::String(uri1.clone()), data]).unwrap();
    let ex = os::object_store_exists(&[Value::String(uri1.clone())]).unwrap();
    assert_eq!(ex, Value::Boolean(true));
    os::object_store_move(&[Value::String(uri1.clone()), Value::String(uri2.clone())]).unwrap();
    let ex_old = os::object_store_exists(&[Value::String(uri1.clone())]).unwrap();
    let ex_new = os::object_store_exists(&[Value::String(uri2.clone())]).unwrap();
    assert_eq!(ex_old, Value::Boolean(false));
    assert_eq!(ex_new, Value::Boolean(true));
}

