use lyra::stdlib::utilities::config::{config_merge, config_get, config_set, config_overlay_env};
use lyra::vm::Value;
use std::collections::HashMap;

fn assoc(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = HashMap::new();
    for (k, v) in pairs { m.insert(k.to_string(), v); }
    Value::Object(m)
}

fn map(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    let mut m = HashMap::new();
    for (k, v) in pairs { m.insert(k.to_string(), v); }
    m
}

#[test]
fn test_deep_merge_and_get_set() {
    let a = assoc(vec![
        ("db", Value::Object(map(vec![("host", Value::String("h1".into()))]))),
        ("log", Value::Integer(1)),
    ]);
    let b = assoc(vec![
        ("db", Value::Object(map(vec![("port", Value::Integer(5432))]))),
        ("log", Value::Integer(2)),
    ]);

    let merged = config_merge(&[a.clone(), b.clone()]).unwrap();
    if let Value::Object(m) = merged {
        // db.host = h1, db.port = 5432, log = 2
        if let Some(Value::Object(db)) = m.get("db") {
            assert_eq!(db.get("host"), Some(&Value::String("h1".into())));
            assert_eq!(db.get("port"), Some(&Value::Integer(5432)));
        } else { panic!("expected db object"); }
        assert_eq!(m.get("log"), Some(&Value::Integer(2)));
    } else { panic!("expected object"); }

    // ConfigSet then ConfigGet
    let empty = assoc(vec![]);
    let set = config_set(&[empty, Value::String("a.b.c".into()), Value::Integer(3)]).unwrap();
    assert_eq!(config_get(&[set.clone(), Value::String("a.b.c".into())]).unwrap(), Value::Integer(3));
    // default when missing
    assert_eq!(config_get(&[set.clone(), Value::String("a.b.d".into()), Value::Integer(9)]).unwrap(), Value::Integer(9));
}

#[test]
fn test_overlay_env_with_prefix() {
    // Set LYRA_DB__HOST=xyz, ensure overlay creates db.host
    std::env::set_var("LYRA_DB__HOST", "xyz");
    let base = assoc(vec![("db", Value::Object(map(vec![("port", Value::Integer(1))])))]);
    let over = config_overlay_env(&[base]).unwrap();
    if let Value::Object(m) = over {
        if let Some(Value::Object(db)) = m.get("db") {
            assert_eq!(db.get("host"), Some(&Value::String("xyz".into())));
            assert_eq!(db.get("port"), Some(&Value::Integer(1)));
        } else { panic!("expected db object"); }
    } else { panic!("expected object"); }
    std::env::remove_var("LYRA_DB__HOST");
}

